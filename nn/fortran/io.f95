module io
    use feature_config
    use feature_util, only : matrix_determinant
    
    implicit none

    integer :: unit_feature_info = 1
    integer :: io_unit_read = 1

    type string_array_type
        character,allocatable :: string(:)
    end type string_array_type

    contains
        subroutine error(routine,message)
            implicit none

            character(len=*),intent(in) :: routine,message
            character,dimension(1:len(routine)+26) :: header
            header(:) = "*"

            write(*,*) '' 
            write(*,*) header
            write(*,*) 'error raised in routine : ',routine
            write(*,*) header
            write(*,*) ''
            write(*,*) 'Error : ',message
            write(*,*) ''
            call exit(0)
        end subroutine

        subroutine unittest_header()
            implicit none

            write(*,*) ""
            write(*,*) "======================="
            write(*,*) "Running Unit Test Suite"
            write(*,*) "======================="
            write(*,*) ""
        end subroutine

        subroutine unittest_summary(tests)
            implicit none

            logical,intent(in) :: tests(:)

            if (all(tests)) then
                write(*,*) ""
                write(*,*) '---------------------------'
                write(*,*) "Unit test summary : SUCCESS"
                write(*,*) '---------------------------'
                write(*,*) ""
            else
                write(*,*) ""
                write(*,*) '---------------------------'
                write(*,*) "Unit test summary : FAILURE"
                write(*,*) '---------------------------'
                write(*,*) ""
            end if
        end subroutine

        subroutine unittest_test(num,success)
            implicit none

            integer,intent(in) :: num
            logical,intent(in) :: success

            if (success) then
                write(*,*) 'test     ',num,'    OK'
            else    
                write(*,*) 'test     ',num,'    FAILED'
            end if
        end subroutine unittest_test

        subroutine info_net()
            use config

            implicit none

            write(*,*) "NN layer    number of nodes"
            write(*,*) "1           ",net_dim%hl1
            write(*,*) "2           ",net_dim%hl2
            write(*,*)

            write(*,*) "Dimension of feature space : ",D
            write(*,*) "Nonlinear function         : ",nlf
            write(*,*)

            write(*,*) "weights layer     shape"
            write(*,*) "1                 ",shape(net_weights%hl1)
            write(*,*) "2                 ",shape(net_weights%hl2)
            write(*,*) "3                 ",shape(net_weights%hl3)
        end subroutine info_net
        
        subroutine info_set(set_type)
            use config

            implicit none

            integer,intent(in) :: set_type

            !* scratch 
            integer :: ii,jj
           
            if ( (set_type.lt.1).or.(set_type.gt.2) ) then
                call error("info_set","unsupported set_type. User error")
            end if

            write(*,*) "========="
            if (set_type.eq.1) then
                write(*,*) "train set"
            else
                write(*,*) "test set"
            end if
            write(*,*) "========="

            do ii=1,data_sets(set_type)%nconf,1
                write(*,*) ""
                write(*,*) "-------------"
                write(*,*) "structure ",ii
                write(*,*) "-------------"
                write(*,*) "features:"
                do jj=1,data_sets(set_type)%configs(ii)%n
                    write(*,*) data_sets(set_type)%configs(ii)%x(:,jj)
                end do
            
                write(*,*) ""
                write(*,*) 'energy : ',data_sets(set_type)%configs(ii)%ref_energy
                write(*,*) "forces:"
                do jj=1,data_sets(set_type)%configs(ii)%n
                    write(*,*) data_sets(set_type)%configs(ii)%ref_fi(:,jj)
                end do

            end do
        end subroutine info_set

        subroutine info_features()
            implicit none

            integer :: ii,jj,dim(1:1),norm_feat(1:2)

            !* idx of normal features
            norm_feat(1) = featureID_StringToInt("acsf_normal-b2")
            norm_feat(2) = featureID_StringToInt("acsf_normal-b3")

            write(*,*) '========'
            write(*,*) 'Features'
            write(*,*) '========'
            do ii=1,feature_params%num_features
                write(*,*) ''   
                write(*,*) 'type     : ',feature_params%info(ii)%ftype
                write(*,*) 'rcut     : ',feature_params%info(ii)%rcut 
                write(*,*) 'fs       : ',feature_params%info(ii)%fs 
                write(*,*) 'rs       : ',feature_params%info(ii)%rs 
                write(*,*) 'eta      : ',feature_params%info(ii)%eta 
                write(*,*) 'xi       : ',feature_params%info(ii)%xi 
                write(*,*) 'lambda   : ',feature_params%info(ii)%lambda
                write(*,*) 'za       : ',feature_params%info(ii)%za
                write(*,*) 'zb       : ',feature_params%info(ii)%zb
                if ( feature_params%info(ii)%ftype.ne.norm_feat(1).and.&
                &feature_params%info(ii)%ftype.ne.norm_feat(2) ) then
                    write(*,*) 'mean     : 0.0'
                    write(*,*) 'prec     : 0.0'
                else
                    dim = shape(feature_params%info(ii)%mean)
                    write(*,*) 'mean     : '
                    do jj=1,dim(1),1
                        write(*,*) feature_params%info(ii)%mean(jj)
                    end do
                    write(*,*) 'prec     : '
                    do jj=1,dim(1),1
                        write(*,*) feature_params%info(ii)%prec(:,jj)
                    end do
                    write(*,*) 'sqrt_det : ',feature_params%info(ii)%sqrt_det
                end if
            end do
        end subroutine

        subroutine read_feature_info(filepath)
            implicit none

            character(len=1024),intent(in) :: filepath

            integer :: num_features,feat_cntr,iostat
            type(string_array_type),allocatable :: split_string(:)
            character(len=1024) :: string

            num_features = read_num_features(filepath)

            if (num_features.eq.0) then
                call error("read_feature_info","no features found in input")
            end if

            allocate(feature_params%info(num_features))
            feature_params%num_features = num_features
write(*,*) 'size = ',size(feature_params%info)
            feat_cntr = 1

            write(*,*) 'have read ',num_features,'features'

            open(unit=unit_feature_info,status='old',file=filepath,action='read')
            do while(.true.)
                read(unit=unit_feature_info,fmt='(a1024)',iostat=iostat) string
                if(iostat.lt.0) then
                    !* eof
                    exit
                end if
                
                if (split(trim(string),":",split_string)) then
                    if ( strings_equal(split_string(1)%string,"feature") ) then
                        call parse_feature(feat_cntr,split_string(2)%string)
                    else if ( strings_equal(split_string(1)%string,"pca") ) then
                        call parse_pca(split_string(2)%string)
                    else if ( strings_equal(split_string(1)%string,"pca_tol") ) then
                        call parse_pca_tol(split_string(2)%string)
                    else
                        call error("read_feature_info","unsupported keyword")
                    end if
                end if
            end do
        end subroutine read_feature_info

        subroutine parse_feature(feat_idx,feat_args)
            implicit none

            integer,intent(in) :: feat_idx
            character,dimension(:),intent(in) :: feat_args
        
            integer :: iostat
            character,dimension(1:1024) :: ftype_string
            character(len=1024) :: tmp
            real(8) :: rcut,fs,eta,xi,lambda,za,zb

            read(feat_args,fmt=*,iostat=iostat) ftype_string 
            read(feat_args,fmt=*,iostat=iostat) tmp
write(*,*) 'input arg :[',feat_args,']'          
            if ( strings_equal(ftype_string,"behler-iso") ) then
                feature_params%info(feat_idx)%ftype = 1

                read(feat_args,fmt=*) rcut,fs,eta,za,zb
            else if ( strings_equal(ftype_string,"behler-ani") ) then
                feature_params%info(feat_idx)%ftype = 2
                
                read(feat_args,fmt=*) rcut,fs,eta,xi,lambda,za,zb
            else
                call error("parse_feature","unsupported feature type") 
            end if
        end subroutine parse_feature

        subroutine parse_pca(feat_args)
            implicit none

            character,dimension(:),intent(in) :: feat_args

            if ( strings_equal(feat_args,"yes") ) then
                feature_params%pca = .true.
            else if ( strings_equal(feat_args,"no") ) then
                feature_params%pca = .false. 
            else
                call error("parse_pca","unsupported value")
            end if
        end subroutine

        subroutine parse_pca_tol(feat_args)
            implicit none
            
            character,dimension(:),intent(in) :: feat_args

            read(feat_args,fmt=*) feature_params%pca_threshold
        end subroutine parse_pca_tol

        integer function read_num_features(filepath)
            implicit none
            
            character(len=1024),intent(in) :: filepath

            integer :: iostat,num_features
            character(len=1024) :: string
            type(string_array_type),allocatable :: split_string(:)

            num_features = 0

            open(unit=unit_feature_info,status='old',file=filepath,action='read')
            do while(.true.)
                read(unit=unit_feature_info,fmt='(a1024)',iostat=iostat) string
                
                if (split(trim(string),":",split_string)) then
                    if (size(split_string).le.1) then
                        call error("read_num_features","missing value for input")
                    end if

                    if ( strings_equal(split_string(1)%string,"feature") ) then
                        num_features = num_features + 1
                    end if

                    deallocate(split_string)
                end if

                if(iostat.lt.0) then
                    !* eof
                    exit
                end if
            end do
            close(unit=unit_feature_info)
            read_num_features = num_features
        end function read_num_features

        logical function strings_equal(string1,string2)
            implicit none
        
            character,dimension(:) :: string1
            character(len=*) :: string2

            integer :: ii
            logical,allocatable :: char_comparison(:)
            logical :: output

            allocate(char_comparison(size(string1)))

            do ii=1,size(string1)
                char_comparison(ii) = all(string1(ii:ii)==string2(ii:ii))
            end do

            output = all(char_comparison)
            deallocate(char_comparison)
write(*,*) 'comparing strings [',string1,'] and [',string2,']'
            strings_equal = output
        end function strings_equal

        logical function split(string,delimator,split_string)
            implicit none

            character(len=*),intent(in) :: string,delimator
            type(string_array_type),intent(inout),allocatable :: split_string(:)

            integer :: ii,char_,intervals(1:2,1:100),cntr
            integer :: string_len,start_idx,end_idx,offset
            logical :: read_line,inside_word
            character(len=1024) :: elem1_

            if (len(adjustl(trim(delimator))).ne.1) then
                call error("split","invalid delimator")
            end if
            intervals = -1
            read_line = .true.

            if (len(adjustl(trim(string))).eq.0) then
                !* blank line
                read_line = .false.
            else
                read(string,*) elem1_

                if (elem1_ == '#') then
                    read_line = .false.
                end if
            end if

            if (read_line) then
                cntr = 0
                
                inside_word = .false.
                do ii=1,len(string)
                    
                    if ( string(ii:ii).ne.delimator(1:1) ) then
                        if (inside_word) then
                            continue
                        else
                            cntr = cntr + 1
                            !* beginning of word
                            intervals(1,cntr) = ii
                            inside_word = .true.
                        end if
                    else
                        if (inside_word) then
                            !* end of word
                            intervals(2,cntr) = ii-1

                            !* now within delimator
                            inside_word = .false.
                        end if
                    end if
                end do
            
                if (intervals(2,cntr).eq.-1) then
                    intervals(2,cntr) = len(trim(string))+1
                end if

                if (cntr.eq.0) then
                    call error("split","delimator not found in line. Input file format error")
                else
                    allocate(split_string(cntr))
                    do ii=1,cntr
                        start_idx = intervals(1,ii)
                        end_idx = intervals(2,ii)
        
                        !* allocate string mem
                        string_len = len(trim(adjustl(string(start_idx:end_idx))))
                        allocate(split_string(ii)%string(string_len))
                        
                        offset = len(adjustl(string(start_idx:end_idx)))-string_len-1
                        
                        do char_=start_idx,end_idx
                            split_string(ii)%string(char_-start_idx+1:char_-start_idx+1) = &
                                    &string(char_+offset:char_+offset)
                        end do
                        write(*,*) 'string element ',ii,' = ',split_string(ii)%string
                    end do
                end if
            end if

            split = read_line
        end function split

        integer function read_natm(file_path)
            implicit none

            character(len=1024),intent(in) :: file_path 

            integer :: line,natm,iostat
            character(len=8) :: string

            line = 1
            natm = 0

            open(unit=io_unit_read,status='old',file=file_path,action='read')
            do while(.true.)
                read(unit=io_unit_read,fmt=*,iostat=iostat) string
                if(line.gt.6) then
                    if(iostat.lt.0) then
                        !* EOF
                        exit
                    end if
                    natm = natm + 1
                end if

                line = line + 1
            end do
            close(unit=io_unit_read)
            !* last two lines are not atom entries
            read_natm = natm - 2
        end function read_natm

        subroutine read_config(set_type,conf,filepath)
            use config
            
            implicit none

            integer,intent(in) :: set_type,conf
            character(len=1024),intent(in) :: filepath

            !* scratch
            integer :: natm,line,iostat
            character(len=8) :: string
            real(8) :: lx,ly,lz,r1,r2,r3,zz,f1,f2,f3
            real(8) :: energy

            !* assumes atom number has already been set and dynamic mem. allocated
            natm = data_sets(set_type)%configs(conf)%n 

            line = 1
            open(unit=io_unit_read,status='old',file=trim(filepath),action='read',iostat=iostat)
            if (iostat.ne.0) then
                call error("read_config","file "//trim(filepath)//" does not exist")
            end if
            
            do while(.true.)
                if (line.eq.1) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string           
                else if ( (line.eq.2).and.(line.le.4) ) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) lx,ly,lz 
                    data_sets(set_type)%configs(conf)%cell(1,line-1) = lx
                    data_sets(set_type)%configs(conf)%cell(2,line-1) = ly
                    data_sets(set_type)%configs(conf)%cell(3,line-1) = lz
                else if( (line.ge.5).and.(line.le.6) ) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string
                    if (iostat.lt.0) then
                        call error("read_config","problem reading configuration")
                    end if
                else if( (line.ge.7).and.(line.le.natm+6) ) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) r1,r2,r3,zz,f1,f2,f3
                    if (iostat.lt.0) then
                        call error("read_config","Unexpected problem reading atoms - check number atoms")
                    end if
                    data_sets(set_type)%configs(conf)%r(1,line-6) = r1
                    data_sets(set_type)%configs(conf)%r(2,line-6) = r2
                    data_sets(set_type)%configs(conf)%r(3,line-6) = r3
                    data_sets(set_type)%configs(conf)%ref_fi(1,line-6) = f1
                    data_sets(set_type)%configs(conf)%ref_fi(2,line-6) = f2
                    data_sets(set_type)%configs(conf)%ref_fi(3,line-6) = f3
                    data_sets(set_type)%configs(conf)%z(line-6) = zz
                    data_sets(set_type)%configs(conf)%r(2,line-6) = r2
                    data_sets(set_type)%configs(conf)%r(3,line-6) = r3
                else if (line.eq.natm+7) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string
                    if (iostat.lt.0) then
                        call error("read_config","problem reading configuration")
                    end if
                else if (line.eq.natm+8) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) energy
                    if (iostat.lt.0) then
                        call error("read_config","Unexpected problem reading energy - check input file")
                    end if
                    data_sets(set_type)%configs(conf)%ref_energy = energy
                else
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string
                    if (iostat.lt.0) then
                        !* EOF as expected
                        exit
                    end if
                end if
            
                line = line + 1
            end do
           
            close(unit=io_unit_read) 
        end subroutine
        
        integer function read_nfeatures(file_path)
            implicit none

            character(len=1024),intent(in) :: file_path 

            integer :: line,natm,iostat
            character(len=80) :: string

            line = 1
            natm = 0

            open(unit=io_unit_read,status='old',file=trim(file_path),action='read')
            do while(.true.)
                read(unit=io_unit_read,fmt=*,iostat=iostat) string
                if(iostat.lt.0) then
                    !* EOF
                    exit
                end if
                line = line + 1
            end do
            close(unit=io_unit_read)
        
            if (line+1.eq.0) then
                call error("read_nfeatures","0 features found in file")
            end if
            
            !* assumes every line is a feature
            read_nfeatures = line - 1 
        end function read_nfeatures

        subroutine read_features(filepath)
            implicit none

            character(len=1024),intent(in) :: filepath
            
            integer :: iostat,line
            character(len=1024) :: string
            real(8) :: rcut,fs,za,zb,mean,prec,xi,eta,lambda,rs
            real(8) :: mean_3(1:3),prec_33(1:3,1:3)

            open(unit=io_unit_read,status='old',file=trim(filepath),action='read',iostat=iostat)
            if (iostat.ne.0) then
                call error("read_config","file "//trim(filepath)//" does not exist")
            end if

            line = 1
            do while(.true.)

                !* parse feature type 
                read(unit=io_unit_read,fmt=*,iostat=iostat) string
                
                if (iostat.lt.0) then
                    !* EOF
                    exit
                end if
                
                feature_params%info(line)%ftype = featureID_StringToInt(trim(string))
                line = line + 1
            end do
            close(unit=io_unit_read)
            
            
            open(unit=io_unit_read,status='old',file=trim(filepath),action='read',iostat=iostat)
            line = 1
            do while(.true.)
                if (feature_params%info(line)%ftype.eq.featureID_StringToInt("atomic_number")) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string
                    if (iostat.ne.0) then
                        call error("read_features","error reading feature entry")
                    end if
                else if (feature_params%info(line)%ftype.eq.featureID_StringToInt("acsf_behler-g1")) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string,rcut,fs,za,zb
                    if (iostat.ne.0) then
                        call error("read_features","error reading feature entry")
                    end if
                    feature_params%info(line)%rcut = rcut
                    feature_params%info(line)%fs = fs
                    feature_params%info(line)%za = za
                    feature_params%info(line)%zb = zb
                else if (feature_params%info(line)%ftype.eq.featureID_StringToInt("acsf_behler-g2")) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string,rcut,fs,eta,rs,za,zb
                    if (iostat.ne.0) then
                        call error("read_features","error reading feature entry")
                    end if
                    feature_params%info(line)%rcut = rcut
                    feature_params%info(line)%fs = fs
                    feature_params%info(line)%eta = eta
                    feature_params%info(line)%rs = rs
                    feature_params%info(line)%za = za
                    feature_params%info(line)%zb = zb
                else if (feature_params%info(line)%ftype.eq.featureID_StringToInt("acsf_behler-g4")) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string,rcut,fs,xi,lambda,eta,za,zb
                    if (iostat.ne.0) then
                        call error("read_features","error reading feature entry")
                    end if
                    feature_params%info(line)%rcut = rcut
                    feature_params%info(line)%fs = fs
                    feature_params%info(line)%xi = xi
                    feature_params%info(line)%lambda = lambda
                    feature_params%info(line)%eta = eta
                    feature_params%info(line)%za = za
                    feature_params%info(line)%zb = zb
                else if (feature_params%info(line)%ftype.eq.featureID_StringToInt("acsf_behler-g5")) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string,rcut,fs,xi,lambda,eta
                    if (iostat.ne.0) then
                        call error("read_features","error reading feature entry")
                    end if
                    feature_params%info(line)%rcut = rcut
                    feature_params%info(line)%fs = fs
                    feature_params%info(line)%xi = xi
                    feature_params%info(line)%lambda = lambda
                    feature_params%info(line)%eta = eta
                    feature_params%info(line)%za = za
                    feature_params%info(line)%zb = zb
                else if (feature_params%info(line)%ftype.eq.featureID_StringToInt("acsf_normal-b2")) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string,rcut,fs,za,zb,mean,prec
                    if (iostat.ne.0) then
                        call error("read_features","error reading feature entry")
                    end if
                    feature_params%info(line)%rcut = rcut
                    feature_params%info(line)%fs = fs
                    feature_params%info(line)%za = za
                    feature_params%info(line)%zb = zb
                    allocate(feature_params%info(line)%mean(1))
                    allocate(feature_params%info(line)%prec(1,1))
                    feature_params%info(line)%mean(1) = mean
                    feature_params%info(line)%prec(1,1) = prec
                    feature_params%info(line)%sqrt_det = sqrt(prec)
                else if (feature_params%info(line)%ftype.eq.featureID_StringToInt("acsf_normal-b3")) then
                    read(unit=io_unit_read,fmt=*,iostat=iostat) string,rcut,fs,za,zb,mean_3,&
                        &prec_33(:,1),prec_33(:,2),prec_33(:,3)
                    if (iostat.ne.0) then
                        call error("read_features","error reading feature entry")
                    end if
                    feature_params%info(line)%rcut = rcut
                    feature_params%info(line)%fs = fs
                    feature_params%info(line)%za = za
                    feature_params%info(line)%zb = zb
                    allocate(feature_params%info(line)%mean(3))
                    allocate(feature_params%info(line)%prec(3,3))
                    feature_params%info(line)%mean(:) = mean_3(:)
                    feature_params%info(line)%prec(:,:) = prec_33(:,:)
                    feature_params%info(line)%sqrt_det = sqrt(matrix_determinant(prec_33))
                else
                    call error("read_features","unexpected feature type in features file.")
                end if
                
                line = line + 1

                if (line.gt.feature_params%num_features) then
                    exit
                end if
            end do
            close(unit=io_unit_read)
        end subroutine read_features
end module io
