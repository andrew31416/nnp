module features
    use config
    use feature_config
    use feature_util

    implicit none

    contains
        subroutine calculate_features()
            implicit none

            integer :: set_type,conf,ii
            real(8),allocatable :: ultra_cart(:,:)
            real(8),allocatable :: ultra_z(:)
            integer,allocatable :: ultra_idx(:)
            real(8) :: mxrcut


            !* max cut off of all interactions
            mxrcut = maxrcut(0)
            
            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    call get_ultracell(maxrcut,1000,set_type,conf,&
                            &ultra_cart,ultra_idx,ultra_z)

                    call calculate_isotropic_info(ultra_cart,ultra_idx)
                end do
            end do

            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    call random_number(data_sets(set_type)%configs(conf)%x(2:,:))

                    data_sets(set_type)%configs(conf)%x(1,:) = 1.0d0
                end do
            end do
            
        end subroutine calculate_features

    

end module
