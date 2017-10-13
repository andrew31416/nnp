module init
    use config

    implicit none
    
    contains
        subroutine initialise_net(num_nodes,nlf_type,feat_D)
            implicit none

            integer,intent(in) :: num_nodes(1:2),nlf_type,feat_D

            !* feature space dimension
            D = feat_D

            !* nodes in each hidden layer
            net_dim%hl1 = num_nodes(1) 
            net_dim%hl2 = num_nodes(2) 

            !* type of nonlinear activation func
            nlf = nlf_type

            !* include bias in weights
            allocate(net_weights%hl1(D+1,net_dim%hl1))
            allocate(net_weights%hl2(net_dim%hl1+1,net_dim%hl2))
            allocate(net_weights%hl3(net_dim%hl2+1))

            allocate(net_units%a%hl1(net_dim%hl1))
            allocate(net_units%a%hl2(net_dim%hl2))

            !* include null value for bias
            allocate(net_units%z%hl1(net_dim%hl1+1))
            allocate(net_units%z%hl2(net_dim%hl2+1))
            
            allocate(net_units%delta%hl1(net_dim%hl1))
            allocate(net_units%delta%hl2(net_dim%hl2))

            !* initialise NN weights
            call random_weights()
            
            call check_input()
        end subroutine initialise_net
   
        subroutine initialise_set(set_type,nconf,ntot,slice_idxs,xin,fin,ein)
            use io, only : error
            
            implicit none

            integer,intent(in) :: set_type  !* 1 = train , 2 = test
            integer,intent(in) :: nconf,ntot,slice_idxs(1:2,1:nconf)
            real(8),intent(in) :: xin(1:D,1:ntot),fin(1:3,1:ntot)
            real(8),intent(in) :: ein(1:nconf)
          
            !* scratch
            integer :: ii,natm,idx1,idx2
            
            if (set_type.eq.1) then
                allocate(train_set(nconf))
            else if (set_type.eq.2) then
                allocate(test_set(nconf))
            else 
                call error("initialise_set","error in specified set")
            end if

            !* parse features
            do ii=1,nconf,1
                !* number atoms
                natm = slice_idxs(2,ii) - slice_idxs(1,ii) + 1

                !* initial and final atoms
                idx1 = slice_idxs(1,ii)
                idx2 = slice_idxs(2,ii)

                if (set_type.eq.1) then
                    train_set(ii)%n = natm
                    
                    !* include null dimension for biases
                    allocate(train_set(ii)%x(D+1,natm))
                    allocate(train_set(ii)%forces(3,natm))

                    !* for biases
                    train_set(ii)%x(1,:) = 1.0d0

                    !* energy
                    train_set(ii)%energy = ein(ii)

                    !* features
                    train_set(ii)%x(2:,1:natm) = xin(:,idx1:idx2)

                    !* forces
                    train_set(ii)%forces(1:3,:) = fin(:,idx1:idx2)
                else
                    test_set(ii)%n = natm
                    
                    !* include null dimension for biases
                    allocate(test_set(ii)%x(D+1,natm))
                    allocate(test_set(ii)%forces(3,natm))

                    !* for biases
                    test_set(ii)%x(1,:) = 1.0d0

                    !* energy
                    test_set(ii)%energy = ein(ii)

                    !* features
                    test_set(ii)%x(2:,1:natm) = xin(:,idx1:idx2)

                    !* forces
                    test_set(ii)%forces(1:3,:) = fin(:,idx1:idx2)
                end if
            end do
        end subroutine

        subroutine random_weights()
            ! ==================================================!
            ! first element of each linear model is the bias    !
            ! ==================================================!
            
            implicit none

            integer :: seed

            !* random seed
            call system_clock(seed)
            
            !* initialise
            call srand(seed)
          
            !* feature weights
            call random_number(net_weights%hl1(2:,:))
            call random_number(net_weights%hl2(2:,:))
            call random_number(net_weights%hl3(2:))

            !* biases
            net_weights%hl1(1,:) = 0.0d0
            net_weights%hl2(1,:) = 0.0d0
            net_weights%hl3(1:)  = 0.0d0

        end subroutine random_weights

        subroutine check_input()
            use io, only : error
            
            implicit none

            !* check nlf type
            if ( (nlf.le.0).or.(nlf.ge.3) ) then
                call error("check_input","nonlinear function type unsupported")
            end if
        end subroutine

        subroutine finalize()
            implicit none

            deallocate(net_weights%hl1)
            deallocate(net_weights%hl2)
            deallocate(net_weights%hl3)
            deallocate(net_units%a%hl1)
            deallocate(net_units%a%hl2)
            deallocate(net_units%z%hl1)
            deallocate(net_units%z%hl2)
            deallocate(net_units%delta%hl1)
            deallocate(net_units%delta%hl2)
            deallocate(train_set)
            deallocate(test_set)
        end subroutine finalize

end module init
